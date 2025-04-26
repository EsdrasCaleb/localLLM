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
        subscription.abovePrice(100.0);
        assertEquals(100.0, subscription.abovePrice());
    }
}
