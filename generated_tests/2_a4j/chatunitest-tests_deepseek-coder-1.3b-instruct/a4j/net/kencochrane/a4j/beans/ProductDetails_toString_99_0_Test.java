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
    public void testGetRatingsImgURL() {
        ProductDetails productDetails = new ProductDetails();
        productDetails.setReviews(new Reviews());
        productDetails.getReviews().setAvgCustomerRating("4.5");
        String expected = "/images/stars-4.gif";
        String actual = productDetails.getRatingsImgURL();
        assertEquals(expected, actual);
    }
}
