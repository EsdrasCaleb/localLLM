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

class ProductDetails_toString_99_1_Test {

    @Test
    void testToString() {
        ProductDetails product = new ProductDetails();
        product.setAsin("12345");
        product.setProductName("Test Product");
        product.setUrl("http://example.com");
        product.setAgeGroup("18+");
        product.setAvailability("In Stock");
        product.setCatalog("Electronics");
        product.setImageUrlSmall("small.jpg");
        product.setImageUrlMedium("medium.jpg");
        product.setImageUrlLarge("large.jpg");
        String expected = "ProductDetails{asin='12345', productName='Test Product', url='http://example.com', ageGroup='18+', availability='In Stock', catalog='Electronics', imageUrlSmall='small.jpg', imageUrlMedium='medium.jpg', imageUrlLarge='large.jpg'}";
        assertEquals(expected, product.toString());
    }
}
