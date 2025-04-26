package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_38_0_Test {

    @Test
    public void testCouponRateBelow() {
        // Arrange
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        double expectedCouponRate = 0.05;
        // Act
        scannerSubscription.couponRateBelow(expectedCouponRate);
        // Assert
        assertEquals(expectedCouponRate, scannerSubscription.couponRateBelow());
    }
}
