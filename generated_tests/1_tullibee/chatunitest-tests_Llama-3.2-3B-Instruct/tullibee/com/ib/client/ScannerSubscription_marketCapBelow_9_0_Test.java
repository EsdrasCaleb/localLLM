package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_marketCapBelow_9_0_Test {

    @Mock
    private ScannerSubscription scannerSubscription;

    @InjectMocks
    private ScannerSubscription testSubject;

    @Test
    public void testMarketCapBelow() {
        // Arrange
        when(scannerSubscription.marketCapBelow()).thenReturn(100.0);
        // Act
        double result = testSubject.marketCapBelow();
        // Assert
        assertEquals(100.0, result, 0.01);
    }

    @Test
    public void testMarketCapBelowWithNoValue() {
        // Arrange
        when(scannerSubscription.marketCapBelow()).thenReturn(Double.POSITIVE_INFINITY);
        // Act
        double result = testSubject.marketCapBelow();
        // Assert
        assertEquals(Double.POSITIVE_INFINITY, result);
    }

    @Test
    public void testMarketCapBelowWithNegativeValue() {
        // Arrange
        when(scannerSubscription.marketCapBelow()).thenReturn(Double.NEGATIVE_INFINITY);
        // Act
        double result = testSubject.marketCapBelow();
        // Assert
        assertEquals(Double.NEGATIVE_INFINITY, result);
    }

    @Test
    public void testMarketCapBelowWithZeroValue() {
        // Arrange
        when(scannerSubscription.marketCapBelow()).thenReturn(0.0);
        // Act
        double result = testSubject.marketCapBelow();
        // Assert
        assertEquals(0.0, result, 0.01);
    }
}
