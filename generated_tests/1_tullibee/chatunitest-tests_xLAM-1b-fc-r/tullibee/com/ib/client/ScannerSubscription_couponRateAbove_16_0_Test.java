package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_16_0_Test {

    @Test
    public void testCouponRateAbove() {
        // Given
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedCouponRate = 1.23;
        subscription.couponRateAbove(expectedCouponRate);
        // When
        double actualCouponRate = subscription.couponRateAbove();
        // Then
        assertEquals(expectedCouponRate, actualCouponRate);
    }
}
