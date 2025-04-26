package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testLocationCode_WhenNotSet_ShouldReturnNull() {
        // Arrange
        // No action needed, m_locationCode is initialized to null by default
        // Act
        String result = scannerSubscription.locationCode();
        // Assert
        assertEquals(null, result);
    }

    @Test
    public void testLocationCode_WhenSet_ShouldReturnCorrectValue() {
        // Arrange
        String expectedLocationCode = "NYC";
        scannerSubscription.locationCode(expectedLocationCode);
        // Act
        String result = scannerSubscription.locationCode();
        // Assert
        assertEquals(expectedLocationCode, result);
    }
}
