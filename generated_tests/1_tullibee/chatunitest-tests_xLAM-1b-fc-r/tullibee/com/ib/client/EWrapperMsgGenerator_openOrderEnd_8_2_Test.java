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

public class EWrapperMsgGenerator_openOrderEnd_8_2_Test {

    @Test
    public void testOpenOrderEnd() {
        String expected = "END OF ORDER MESSAGE";
        String result = EWrapperMsgGenerator.openOrderEnd();
        assertEquals(expected, result);
    }
}
