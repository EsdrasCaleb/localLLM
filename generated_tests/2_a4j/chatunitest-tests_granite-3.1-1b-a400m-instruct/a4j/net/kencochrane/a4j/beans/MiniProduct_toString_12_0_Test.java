package net.kencochrane.a4j.beans;

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
        MiniProduct miniProduct = new MiniProduct();
        miniProduct.setName("Product Name");
        miniProduct.setManufacturer("Manufacturer");
        miniProduct.setPrice("9.99");
        miniProduct.setAsin("B08P1Z191K");
        miniProduct.setImageURL("https://example.com/image.jpg");
        miniProduct.setProductUrl("https://example.com/product");
        String expectedString = "Product Name \nManufacturer: Manufacturer \nPrice: 9.99 \nImage URL: https://example.com/image.jpg \nProduct URL: https://example.com/product";
        String actualString = miniProduct.toString();
        assertEquals(expectedString, actualString);
    }
}
