package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_maturityDateAbove_14_4_Test {

    @Test
    void testMaturityDateAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateAbove("2022-01-01");
        assertEquals("2022-01-01", subscription.maturityDateAbove());
    }
}
