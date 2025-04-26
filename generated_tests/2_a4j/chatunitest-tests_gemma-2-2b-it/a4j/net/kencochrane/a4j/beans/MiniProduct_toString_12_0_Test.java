package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class MiniProduct_toString_12_0_Test {

    @Test
    void testToString() {
        MiniProduct product = new MiniProduct();
        product.setName("Test Product");
        product.setManufacturer("Test Manufacturer");
        product.setPrice("10.99");
        product.setAsin("B000000000");
        product.setImageURL("https://www.example.com/image.jpg");
        product.setProductUrl("https://www.example.com/product");
        String expectedString = "B000000000 \n Test Product \n Test Manufacturer \n 10.99 \n https://www.example.com/image.jpg";
        String actualString = product.toString();
        assertEquals(expectedString, actualString);
    }
}
