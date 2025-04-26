package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_maturityDateBelow_36_0_Test {

    @ParameterizedTest
    @ValueSource(strings = { "2024-10-26", "2023-03-15", "" })
    void testMaturityDateBelow(String maturityDate) {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateBelow(maturityDate);
        try {
            Field maturityDateBelowField = ScannerSubscription.class.getDeclaredField("m_maturityDateBelow");
            maturityDateBelowField.setAccessible(true);
            assertEquals(maturityDate, maturityDateBelowField.get(subscription));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing field: " + e.getMessage());
        }
    }
}
