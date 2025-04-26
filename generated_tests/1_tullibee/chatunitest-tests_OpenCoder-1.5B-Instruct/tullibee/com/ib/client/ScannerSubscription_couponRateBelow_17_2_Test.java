package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_17_2_Test {

    ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);

    @Test
    public void testCouponRateBelow() {
        // Arrange
        double expected = 0.05;
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Mockito.when(scannerSubscription.couponRateBelow()).thenReturn(expected);
        // Act
        double actual = scannerSubscription.couponRateBelow();
        // Assert
        assertEquals(expected, actual);
    }
}
