package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class MiniProduct_toString_12_0_Test {

    private MiniProduct miniProduct;

    @BeforeEach
    void setUp() {
        miniProduct = new MiniProduct();
    }

    @Test
    void testToString() throws NoSuchFieldException, IllegalAccessException {
        // Set up test data using reflection
        Field asinField = MiniProduct.class.getDeclaredField("asin");
        asinField.setAccessible(true);
        asinField.set(miniProduct, "B08N5WRWNW");
        Field nameField = MiniProduct.class.getDeclaredField("name");
        nameField.setAccessible(true);
        nameField.set(miniProduct, "Wireless Bluetooth Headphones");
        Field manufacturerField = MiniProduct.class.getDeclaredField("manufacturer");
        manufacturerField.setAccessible(true);
        manufacturerField.set(miniProduct, "ExampleCorp");
        Field priceField = MiniProduct.class.getDeclaredField("price");
        priceField.setAccessible(true);
        priceField.set(miniProduct, "$29.99");
        Field imageURLField = MiniProduct.class.getDeclaredField("imageURL");
        imageURLField.setAccessible(true);
        imageURLField.set(miniProduct, "http://example.com/image.jpg");
        // Expected output
        String expectedOutput = "B08N5WRWNW \n Wireless Bluetooth Headphones \n ExampleCorp \n $29.99 \n http://example.com/image.jpg";
        // Actual output from the toString() method
        String actualOutput = miniProduct.toString();
        // Assert that the actual output matches the expected output
        assertEquals(expectedOutput, actualOutput);
    }
}
