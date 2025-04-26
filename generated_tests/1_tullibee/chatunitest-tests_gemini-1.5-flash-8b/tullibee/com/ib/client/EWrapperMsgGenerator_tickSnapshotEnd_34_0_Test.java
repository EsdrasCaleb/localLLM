package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_tickSnapshotEnd_34_0_Test {

    @ParameterizedTest
    @ValueSource(ints = { 1, 10, 100, -5 })
    void testTickSnapshotEnd(int tickerId) {
        String expected = "id=" + tickerId + " =============== end ===============";
        String actual = EWrapperMsgGenerator.tickSnapshotEnd(tickerId);
        assertEquals(expected, actual);
    }
}
