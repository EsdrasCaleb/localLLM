package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UNKNOWN_ID;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.Vector;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
public class EReader_stop_12_0_Test {

    @InjectMocks
    private EReader reader;

    @Test
    public void testStop() {
        assertDoesNotThrow(() -> reader.stop());
    }

    @Test
    public void testStopThrowsException() {
        assertThrows(NullPointerException.class, () -> reader.stop());
    }
}
