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

class ProductDetails_toString_99_0_Test {

    @Test
    public void testToString() throws Exception {
        // Create a mock instance of ProductDetails
        ProductDetails productDetails = spy(ProductDetails.class);
        // Define some sample values for the fields
        when(productDetails.getAsin()).thenReturn("B0789ABC");
        when(productDetails.getProductName()).thenReturn("Example Product");
        when(productDetails.getCatalog()).thenReturn("Online Catalog");
        when(productDetails.getReleaseDate()).thenReturn("2023-01-01");
        when(productDetails.getManufacturer()).thenReturn("Manufacturer Inc.");
        when(productDetails.getImageUrlSmall()).thenReturn("small.jpg");
        when(productDetails.getImageUrlMedium()).thenReturn("medium.jpg");
        when(productDetails.getImageUrlLarge()).thenReturn("large.jpg");
        when(productDetails.getMedia()).thenReturn("Video");
        when(productDetails.getIsbn()).thenReturn("ISBN1234567890");
        when(productDetails.getAvailability()).thenReturn("In Stock");
        when(productDetails.getMpn()).thenReturn("MPN123456");
        when(productDetails.getListPrice()).thenReturn("$19.99");
        when(productDetails.getOurPrice()).thenReturn("$19.99");
        // Call the toString() method on the mock instance
        String result = productDetails.toString();
        // Assert that the result contains all the expected fields
        assertTrue(result.contains("ASIN: B0789ABC"));
        assertTrue(result.contains("Product Name: Example Product"));
        assertTrue(result.contains("Catalog: Online Catalog"));
        assertTrue(result.contains("Release Date: 2023-01-01"));
        assertTrue(result.contains("Manufacturer: Manufacturer Inc."));
        assertTrue(result.contains("Image URL Small: small.jpg"));
        assertTrue(result.contains("Image URL Medium: medium.jpg"));
        assertTrue(result.contains("Image URL Large: large.jpg"));
        assertTrue(result.contains("Media: Video"));
        assertTrue(result.contains("ISBN: ISBN1234567890"));
        assertTrue(result.contains("Availability: In Stock"));
        assertTrue(result.contains("MPN: MPN123456"));
        assertTrue(result.contains("List Price: $19.99"));
        assertTrue(result.contains("Our Price: $19.99"));
    }
}
