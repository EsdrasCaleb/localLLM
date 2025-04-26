package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_4_2_Test {

    @Test
    public void testAbovePrice() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(Double.MAX_VALUE, subscription.abovePrice(), 0.0001);
    }
}
