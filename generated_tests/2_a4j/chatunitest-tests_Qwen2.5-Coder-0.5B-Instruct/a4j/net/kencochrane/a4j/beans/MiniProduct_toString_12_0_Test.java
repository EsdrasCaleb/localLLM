package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class MiniProduct_toString_12_0_Test {

    @Test
    void testToString() {
        // Create a mock instance of MiniProduct
        MiniProduct miniProduct = mock(MiniProduct.class);
        // Set the expected values for the fields
        when(miniProduct.getAsin()).thenReturn("12345");
        when(miniProduct.getName()).thenReturn("Product A");
        when(miniProduct.getManufacturer()).thenReturn("Brand A");
        when(miniProduct.getPrice()).thenReturn("19.99");
        when(miniProduct.getImageURL()).thenReturn("https://example.com/image.jpg");
        when(miniProduct.getProductUrl()).thenReturn("http://example.com/product.html");
        // Create an instance of MiniProduct
        MiniProduct actualMiniProduct = new MiniProduct();
        // Call the toString() method on the mock object
        String result = actualMiniProduct.toString();
        // Verify that the result matches the expected string
        assertEquals("12345 Product A Brand A 19.99 https://example.com/image.jpg http://example.com/product.html", result);
    }
}
