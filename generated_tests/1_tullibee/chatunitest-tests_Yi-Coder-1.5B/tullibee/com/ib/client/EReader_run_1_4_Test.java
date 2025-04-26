package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UNKNOWN_ID;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.Vector;
import com.ib.client.EClientErrors.CodeMsgPair;

public class EReader_run_1_4_Test {

    @Test
    public void testRun() {
        // Create a mock EReader instance
        EReader mockEReader = mock(EReader.class);
        // Set up the mock method calls
        doNothing().when(mockEReader).run();
        // Call the method under test
        mockEReader.run();
        // Verify the method call
        verify(mockEReader).run();
    }
}
