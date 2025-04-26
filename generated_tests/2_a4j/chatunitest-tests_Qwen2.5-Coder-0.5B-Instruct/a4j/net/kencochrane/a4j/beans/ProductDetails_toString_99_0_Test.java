package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;
import java.math.BigDecimal;
import java.text.DecimalFormat;

public class ProductDetails_toString_99_0_Test {

    @Test
    public void testGetSavings() {
        ProductDetails productDetails = new ProductDetails();
        String savings = productDetails.getSavings();
        assertNotNull(savings);
        assertEquals(" (You save $20 that's 20% off the list price!)", savings);
    }

    @Test
    public void testGetRatingsImgURL() {
        ProductDetails productDetails = new ProductDetails();
        String imgURL = productDetails.getRatingsImgURL();
        assertNotNull(imgURL);
        assertEquals("/images/stars-20.gif", imgURL);
    }

    @Test
    public void testGetRecommendation() {
        ProductDetails productDetails = new ProductDetails();
        String recommendation = productDetails.getRecommendation();
        assertNotNull(recommendation);
        assertEquals("20 of our Customers recommend this Product!", recommendation);
    }
}
