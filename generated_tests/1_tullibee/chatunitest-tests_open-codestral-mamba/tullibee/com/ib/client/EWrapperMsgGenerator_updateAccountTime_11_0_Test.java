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

public class EWrapperMsgGenerator_updateAccountTime_11_0_Test {

    @Test
    public void testUpdateAccountTime() {
        String timeStamp = "2021-09-01 10:00:00";
        String expected = "updateAccountTime: " + timeStamp;
        String actual = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        assertEquals(expected, actual);
    }
}
