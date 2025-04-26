package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickSnapshotEnd_34_0_Test {

    @Test
    public void testTickSnapshotEnd() {
        int tickerId = 123;
        String expectedMessage = "id=123 =============== end ===============";
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        String result = generator.tickSnapshotEnd(tickerId);
        assertEquals(expectedMessage, result);
    }
}
