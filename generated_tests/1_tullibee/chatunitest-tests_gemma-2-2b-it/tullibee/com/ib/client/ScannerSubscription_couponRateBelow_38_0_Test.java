package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_38_0_Test {

    @Test
    void couponRateBelow() {
        ScannerSubscription instance = new ScannerSubscription();
        instance.couponRateBelow(0.01);
        assertEquals(0.01, instance.couponRateBelow());
    }
}
