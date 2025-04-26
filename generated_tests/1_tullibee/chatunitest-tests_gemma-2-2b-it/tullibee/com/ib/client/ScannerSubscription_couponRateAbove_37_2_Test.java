package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_37_2_Test {

    @Test
    void couponRateAbove_Test() {
        ScannerSubscription sut = new ScannerSubscription();
        // Set up the test values
        double r = 0.10;
        // Invoke the method being tested
        sut.couponRateAbove(r);
        // Assert the result
        assertEquals(r, sut.couponRateAbove(), 0.001);
    }
}
