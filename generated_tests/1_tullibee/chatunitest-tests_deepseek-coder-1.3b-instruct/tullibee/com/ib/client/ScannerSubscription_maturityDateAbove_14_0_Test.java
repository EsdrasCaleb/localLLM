package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_maturityDateAbove_14_0_Test {

    @Test
    void testMaturityDateAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateAbove("2022-12-31");
        assertEquals("2022-12-31", subscription.maturityDateAbove());
    }
}
