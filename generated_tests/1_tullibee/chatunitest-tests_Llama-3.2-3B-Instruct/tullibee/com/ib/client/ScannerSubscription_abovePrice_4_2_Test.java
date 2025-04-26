// ScannerSubscription_abovePrice_4_2_Test.java
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_abovePrice_4_2_Test {

    @Mock
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    void setup() {
        when(scannerSubscription.abovePrice()).thenReturn(10.0);
    }

    @Test
    public void testAbovePrice_ReturnsAbovePrice() {
        // Act
        double result = scannerSubscription.abovePrice();
        // Assert
        assertEquals(10.0, result, 0.0);
    }

    @Test
    public void testAbovePrice_ReturnsDefaultAbovePrice() {
        // Arrange
        scannerSubscription = new ScannerSubscription();
        // Act
        double result = scannerSubscription.abovePrice();
        // Assert
        assertEquals(Double.MAX_VALUE, result, 0.0);
    }

    @Test
    public void testAbovePrice_WithNullScannerSubscription_ThrowsNullPointerException() {
        // Act and Assert
        assertThrows(NullPointerException.class, () -> scannerSubscription.abovePrice());
    }
}
