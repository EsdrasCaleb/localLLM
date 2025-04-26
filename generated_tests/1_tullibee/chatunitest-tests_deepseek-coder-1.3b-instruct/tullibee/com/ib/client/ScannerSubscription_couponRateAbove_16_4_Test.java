package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_16_4_Test {

    @Test
    public void testCouponRateAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.couponRateAbove(10.5);
        assertEquals(10.5, subscription.couponRateAbove());
    }
}
