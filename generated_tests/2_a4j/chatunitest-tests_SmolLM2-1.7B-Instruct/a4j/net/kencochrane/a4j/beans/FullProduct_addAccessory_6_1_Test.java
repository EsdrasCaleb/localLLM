package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class FullProduct_addAccessory_6_1_Test {

    @Test
    public void testAddAccessory() {
        // Arrange
        FullProduct focal = new FullProduct();
        MiniProduct miniProduct = new MiniProduct();
        // Act
        focal.addAccessory(miniProduct);
        // Assert
        assertNotNull(focal.getAccessories());
        assertEquals(1, focal.getAccessories().size());
    }
}
