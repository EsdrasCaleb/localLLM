package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_execDetailsEnd_20_0_Test {

    @Test
    void testExecDetailsEnd() {
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        int requestId = 1234;
        String expected = "FA: " + requestId;
        String result = generator.execDetailsEnd(requestId);
        assertEquals(expected, result);
    }
}
