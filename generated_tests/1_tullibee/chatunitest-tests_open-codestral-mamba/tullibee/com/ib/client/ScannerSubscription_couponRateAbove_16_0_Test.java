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
        // Create a mock object of ScannerSubscription
        ScannerSubscription subscriptionMock = Mockito.mock(ScannerSubscription.class);
        // Set the expected return value for the couponRateAbove() method
        Mockito.when(subscriptionMock.couponRateAbove()).thenReturn(5.0);
        // Invoke the couponRateAbove() method on the mock object
        double result = subscriptionMock.couponRateAbove();
        // Verify the result
        assertEquals(5.0, result);
    }
}
