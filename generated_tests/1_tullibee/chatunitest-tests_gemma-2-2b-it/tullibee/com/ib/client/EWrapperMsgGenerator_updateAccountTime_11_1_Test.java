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

public class EWrapperMsgGenerator_updateAccountTime_11_1_Test {

    @Test
    void testUpdateAccountTime() {
        String timeStamp = "2023-10-26 10:00:00";
        String actualResult = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        assertEquals("updateAccountTime: 2023-10-26 10:00:00", actualResult);
    }
}
