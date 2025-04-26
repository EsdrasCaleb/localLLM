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
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedCouponRate = 0.01;
        subscription.couponRateAbove(expectedCouponRate);
        double actualCouponRate = subscription.couponRateAbove();
        assertEquals(expectedCouponRate, actualCouponRate);
    }
}
