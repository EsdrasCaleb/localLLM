package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_15_2_Test {

    @Test
    public void testMaturityDateBelow() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateBelow("2022-12-31");
        assertEquals("2022-12-31", subscription.maturityDateBelow());
    }
}
