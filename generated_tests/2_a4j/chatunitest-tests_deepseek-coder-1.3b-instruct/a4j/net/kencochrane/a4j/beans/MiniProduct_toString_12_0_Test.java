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
        miniProduct.setAsin("123");
        miniProduct.setName("Product Name");
        miniProduct.setManufacturer("Product Manufacturer");
        miniProduct.setPrice("Product Price");
        miniProduct.setImageURL("Product Image URL");
        miniProduct.setProductUrl("Product URL");
        String expected = "123 \n Product Name \n Product Manufacturer \n Product Price \n Product Image URL";
        assertEquals(expected, miniProduct.toString());
    }
}
