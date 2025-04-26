package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.text.DecimalFormat;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;

public class ProductDetails_toString_99_0_Test {

    @Test
    void testGetSavings_withValidPrices() {
        ProductDetails pd = new ProductDetails();
        pd.setListPrice("$10.00");
        pd.setOurPrice("$8.00");
        String savings = pd.getSavings();
        assertEquals(" (You save $2.00 that's 20.00% off the list price!)", savings);
    }

    @Test
    void testGetSavings_withNullListPrice() {
        ProductDetails pd = new ProductDetails();
        pd.setOurPrice("$8.00");
        String savings = pd.getSavings();
        assertNull(savings);
    }

    @Test
    void testGetSavings_withNullOurPrice() {
        ProductDetails pd = new ProductDetails();
        pd.setListPrice("$10.00");
        String savings = pd.getSavings();
        assertNull(savings);
    }

    @Test
    void testGetSavings_withInvalidPrices() {
        ProductDetails pd = new ProductDetails();
        pd.setListPrice("abc");
        pd.setOurPrice("def");
        String savings = pd.getSavings();
        assertNull(savings);
    }

    @Test
    void testGetSavings_withZeroSavings() {
        ProductDetails pd = new ProductDetails();
        pd.setListPrice("$10.00");
        pd.setOurPrice("$10.00");
        String savings = pd.getSavings();
        assertEquals(" (You save $0.00 that's 0.00% off the list price!)", savings);
    }

    @Test
    void testGetRatingsImgURL_withValidRating() throws Exception {
        ProductDetails pd = new ProductDetails();
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("4.5");
        pd.setReviews(reviews);
        String url = pd.getRatingsImgURL();
        assertEquals("/images/stars-4.gif", url);
    }

    @Test
    void testGetRatingsImgURL_withNullReviews() {
        ProductDetails pd = new ProductDetails();
        String url = pd.getRatingsImgURL();
        assertNull(url);
    }

    @Test
    void testGetRatingsImgURL_withInvalidRating() throws Exception {
        ProductDetails pd = new ProductDetails();
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("abc");
        pd.setReviews(reviews);
        String url = pd.getRatingsImgURL();
        assertNull(url);
    }

    @Test
    void testGetRatingsImgURL_withRatingOutOfRange() throws Exception {
        ProductDetails pd = new ProductDetails();
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("6.5");
        pd.setReviews(reviews);
        String url = pd.getRatingsImgURL();
        assertNull(url);
    }

    @Test
    void testGetRecommendation_withValidRating() throws Exception {
        ProductDetails pd = new ProductDetails();
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("4");
        pd.setReviews(reviews);
        String recommendation = pd.getRecommendation();
        assertEquals("80% of our Customers recommend this Product!", recommendation);
    }

    @Test
    void testGetRecommendation_withNullReviews() {
        ProductDetails pd = new ProductDetails();
        String recommendation = pd.getRecommendation();
        assertNull(recommendation);
    }

    @Test
    void testGetRecommendation_withInvalidRating() throws Exception {
        ProductDetails pd = new ProductDetails();
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("abc");
        pd.setReviews(reviews);
        String recommendation = pd.getRecommendation();
        assertNull(recommendation);
    }

    @Test
    void testGetRecommendation_withRatingOutOfRange() throws Exception {
        ProductDetails pd = new ProductDetails();
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("6.5");
        pd.setReviews(reviews);
        String recommendation = pd.getRecommendation();
        assertNull(recommendation);
    }
}
