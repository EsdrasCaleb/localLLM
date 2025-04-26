package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_0_Test {

    private ScannerSubscription scannerSubscription;

    @Test
    public void testMarketCapAbove() {
        // Arrange
        scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Mockito.when(scannerSubscription.marketCapAbove()).thenReturn(Double.MAX_VALUE);
        // Act
        double result = scannerSubscription.marketCapAbove();
        // Assert
        assertEquals(Double.MAX_VALUE, result);
    }
}
