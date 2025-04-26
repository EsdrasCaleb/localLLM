package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class MiniProduct_toString_12_0_Test {

    private MiniProduct miniProduct;

    @BeforeEach
    void setUp() {
        miniProduct = new MiniProduct();
    }

    @Test
    void testToString_validData() throws NoSuchFieldException, IllegalAccessException {
        miniProduct.setAsin("B001234567");
        miniProduct.setName("Test Product");
        miniProduct.setManufacturer("Example Manufacturer");
        miniProduct.setPrice("19.99");
        miniProduct.setImageURL("https://example.com/image.jpg");
        String expected = "B001234567 \n Test Product \n Example Manufacturer \n 19.99 \n https://example.com/image.jpg";
        assertEquals(expected, miniProduct.toString());
    }

    @Test
    void testToString_emptyValues() throws NoSuchFieldException, IllegalAccessException {
        String expected = "null \n null \n null \n null \n null";
        assertEquals(expected, new MiniProduct().toString());
    }

    @Test
    void testToString_nullValues() throws NoSuchFieldException, IllegalAccessException {
        miniProduct.setAsin(null);
        miniProduct.setName(null);
        miniProduct.setManufacturer(null);
        miniProduct.setPrice(null);
        miniProduct.setImageURL(null);
        String expected = "null \n null \n null \n null \n null";
        assertEquals(expected, miniProduct.toString());
    }

    @Test
    void testToString_specificValues() throws NoSuchFieldException, IllegalAccessException {
        miniProduct.setAsin("123");
        miniProduct.setName("Product 1");
        miniProduct.setManufacturer("Manu 1");
        miniProduct.setPrice("10");
        miniProduct.setImageURL("image1.jpg");
        String expected = "123 \n Product 1 \n Manu 1 \n 10 \n image1.jpg";
        assertEquals(expected, miniProduct.toString());
    }
}
