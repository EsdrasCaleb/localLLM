package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_5_0_Test {

    @Test
    public void belowPriceTest() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.belowPrice(100.00);
        double result = subscription.belowPrice();
        assertEquals(100.00, result);
    }
}
