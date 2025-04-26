package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_17_0_Test {

    @Test
    public void testCouponRateBelow() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedCouponRateBelow = 1.0;
        // Act
        double actualCouponRateBelow = subscription.couponRateBelow();
        // Assert
        assertEquals(expectedCouponRateBelow, actualCouponRateBelow);
    }
}
