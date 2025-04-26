package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_marketCapAbove_29_1_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @Test
    public void testMarketCapAbove_SetMarketCapAbove() {
        // Arrange
        double marketCapAbove = 100.0;
        // Act
        scannerSubscription.marketCapAbove(marketCapAbove);
        // Assert
        assertEquals(marketCapAbove, scannerSubscription.marketCapAbove());
    }

    @Test
    public void testMarketCapAbove_SetMarketCapAboveWithExistingValue() {
        // Arrange
        double marketCapAbove = 100.0;
        scannerSubscription.marketCapAbove(marketCapAbove);
        // Act
        scannerSubscription.marketCapAbove(marketCapAbove);
        // Assert
        assertEquals(marketCapAbove, scannerSubscription.marketCapAbove());
    }

    @Test
    public void testMarketCapAbove_SetMarketCapAboveWithNegativeValue() {
        // Arrange
        double marketCapAbove = -100.0;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> scannerSubscription.marketCapAbove(marketCapAbove));
    }

    @Test
    public void testMarketCapAbove_SetMarketCapAboveWithDoubleNegativeValue() {
        // Arrange
        double marketCapAbove = Double.NEGATIVE_INFINITY;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> scannerSubscription.marketCapAbove(marketCapAbove));
    }

    @Test
    public void testMarketCapAbove_SetMarketCapAboveWithDoublePositiveInfinity() {
        // Arrange
        double marketCapAbove = Double.POSITIVE_INFINITY;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> scannerSubscription.marketCapAbove(marketCapAbove));
    }
}
