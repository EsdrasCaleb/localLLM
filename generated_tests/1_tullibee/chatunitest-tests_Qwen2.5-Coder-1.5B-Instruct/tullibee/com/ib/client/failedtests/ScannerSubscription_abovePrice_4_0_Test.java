package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_4_0_Test {

    @Test
    public void testAbovePriceReturnsMaxPriceSpecified() {
        // Arrange
        ScannerSubscription scannerSubscription = mock(ScannerSubscription.class);
        when(scannerSubscription.abovePrice()).thenReturn(200.5);
        // Act
        double result = scannerSubscription.abovePrice();
        // Assert
        assertEquals(200.5, result);
    }

    @Test
    public void testAbovePriceReturnsDefaultValueWhenNotSet() {
        // Arrange
        ScannerSubscription scannerSubscription = mock(ScannerSubscription.class);
        // Use null as default value
        when(scannerSubscription.abovePrice()).thenReturn(null);
        // Act
        double result = scannerSubscription.abovePrice();
        // Assert
        assertEquals(Double.MAX_VALUE, result);
    }
}
