package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_4_0_Test {

    @Test
    public void testAbovePrice() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(1);
        subscription.instrument("ABC");
        subscription.locationCode("US");
        subscription.scanCode("12345");
        subscription.belowPrice(10.0);
        subscription.abovePrice();
        assertEquals(Double.MAX_VALUE, subscription.abovePrice());
    }
}
