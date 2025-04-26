package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class FullProduct_addAccessory_6_0_Test {

    @Test
    public void testAddAccessory() {
        // Arrange
        FullProduct fullProduct = new FullProduct();
        MiniProduct product = new MiniProduct();
        // Act
        fullProduct.addAccessory(product);
        // Assert
        fullProduct.getAccessories().add(product);
        System.out.println("Accessory list: " + fullProduct.getAccessories().toString());
    }
}
