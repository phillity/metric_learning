import tensorflow as tf


__all__ = [
    "hardest_contrastive_loss", "contrastive_loss"
]

"""
Dimensionality Reduction by Learning an Invariant Mapping (CVPR '06)
Raia Hadsell, Sumit Chopra, Yann LeCun
http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Deep Learning Face Representation by Joint Identification-Verification (NIPS '14)
Yi Sun, Xiaogang Wang, Xiaoou Tang
https://arxiv.org/abs/1406.4773

contrastive loss:
L = {(1/2) * ||f(x_i) - f(x_j)||^2 if label[i]==label[j]
    {(1/2) * max(0, margin - ||f(x_i) - f(x_j)||)^2 if label[i]!=label[j]
"""


def pairwise_distances(embeddings, squared=True):
    # get pariwise-distance matrix
    # p_dist_mat[i, j] = ||f(x_i) - f(x_j)||^2
    #                  = ||f(x_i)||^2  - 2 <f(x_i), f(x_j)> + ||f(x_j)||^2
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    p_dist_mat = tf.expand_dims(
        square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    p_dist_mat = tf.maximum(p_dist_mat, 0.0)

    if not squared:
        mask = tf.cast(tf.equal(p_dist_mat, 0.0), tf.float32)
        p_dist_mat = p_dist_mat + tf.multiply(mask, 1e-16)
        p_dist_mat = tf.sqrt(p_dist_mat)
        p_dist_mat = p_dist_mat * (1.0 - mask)

    return p_dist_mat


def positive_pair_mask(labels):
    diag = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    zero_diag = tf.logical_not(diag)
    pos_mask = tf.equal(labels, tf.transpose(labels))
    pos_mask = tf.linalg.band_part(pos_mask, 0, -1)
    pos_mask = tf.logical_and(pos_mask, zero_diag)
    return pos_mask


def negative_pair_mask(labels):
    neg_mask = tf.logical_not(tf.equal(labels, tf.transpose(labels)))
    neg_mask = tf.linalg.band_part(neg_mask, 0, -1)
    return neg_mask


def hardest_contrastive_loss(labels, embeddings, margin=1.0):
    # get pairwise distances of embeddings
    pairwise_dist = pairwise_distances(embeddings, squared=False)

    # get positive pair mask
    pos_mask = positive_pair_mask(labels)
    pos_mask = tf.cast(pos_mask, tf.float32)

    # Get average of all hardest positive pair distances
    pos_distances = tf.multiply(tf.square(pairwise_dist), pos_mask)
    pos_distances = tf.reduce_max(pos_distances, axis=1, keepdims=True)
    num_pos_distances = tf.reduce_sum(
        tf.cast(tf.logical_not(tf.equal(pos_distances, 0.0)), tf.float32))
    pos_distances = tf.reduce_sum(pos_distances)

    # get negative pair mask
    neg_mask = negative_pair_mask(labels)
    neg_mask = tf.cast(neg_mask, tf.float32)

    # get average of all hardest negative pair distances (only considering those less than margin)
    neg_distances = tf.multiply(pairwise_dist, neg_mask)
    neg_distances = tf.square(tf.maximum(
        0.0, tf.multiply(margin, neg_mask) - neg_distances))
    neg_distances = tf.reduce_max(neg_distances, axis=1, keepdims=True)
    num_neg_distances = tf.reduce_sum(
        tf.cast(tf.logical_not(tf.equal(neg_distances, 0.0)), tf.float32))
    neg_distances = tf.reduce_sum(neg_distances)

    # get number of non-zero triplets and normalize
    contrastive_loss = (pos_distances + neg_distances) / \
        (num_pos_distances + num_neg_distances + 1e-16)
    return contrastive_loss


def contrastive_loss(labels, embeddings, margin=1.0):
    # get pairwise distances of embeddings
    pairwise_dist = pairwise_distances(embeddings, squared=False)

    # get positive pair mask
    pos_mask = positive_pair_mask(labels)
    pos_mask = tf.cast(pos_mask, tf.float32)

    # Get average of all positive pair distances
    pos_distances = tf.multiply(tf.square(pairwise_dist), pos_mask)
    num_pos_distances = tf.reduce_sum(pos_mask)
    pos_distances = tf.reduce_sum(pos_distances)

    # get negative pair mask
    neg_mask = negative_pair_mask(labels)
    neg_mask = tf.cast(neg_mask, tf.float32)

    # get average of all negative pair distances (only considering those less than margin)
    neg_distances = tf.multiply(pairwise_dist, neg_mask)
    neg_distances = tf.square(tf.maximum(
        0.0, tf.multiply(margin, neg_mask) - neg_distances))
    num_neg_distances = tf.reduce_sum(
        tf.cast(tf.logical_not(tf.equal(neg_distances, 0.0)), tf.float32))
    neg_distances = tf.reduce_sum(neg_distances)

    # get total contrastive loss and normalize
    contrastive_loss = (pos_distances + neg_distances) / \
        (num_pos_distances + num_neg_distances + 1e-16)
    return contrastive_loss
