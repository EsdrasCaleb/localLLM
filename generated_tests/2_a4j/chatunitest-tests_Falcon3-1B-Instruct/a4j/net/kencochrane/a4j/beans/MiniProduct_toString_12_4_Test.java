package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class MiniProduct_toString_12_4_Test {

    @Test
    public void testToString() {
        // Arrange
        MiniProduct miniProduct = new MiniProduct();
        // Act
        String expectedString = "asin \n " + miniProduct.name + " \n " + miniProduct.manufacturer + " \n " + miniProduct.price + " \n " + miniProduct.imageURL;
        // Assert
        Assertions.assertTrue(expectedString.equals(miniProduct.toString()), "Expected 'asin \n " + miniProduct.name + " \n " + miniProduct.manufacturer + " \n " + miniProduct.price + " \n " + miniProduct.imageURL);
    }
}
