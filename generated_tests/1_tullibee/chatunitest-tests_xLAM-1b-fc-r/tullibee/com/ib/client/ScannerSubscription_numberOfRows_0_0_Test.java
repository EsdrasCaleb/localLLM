package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_numberOfRows_0_0_Test {

    @Test
    void testNumberOfRows() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        assertEquals(10, subscription.numberOfRows());
    }
}
