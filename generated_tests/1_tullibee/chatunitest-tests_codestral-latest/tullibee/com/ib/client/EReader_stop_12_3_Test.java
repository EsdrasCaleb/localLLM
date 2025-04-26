package com.ib.client;

import java.io.DataInputStream;
import java.io.IOException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UNKNOWN_ID;
import java.util.Vector;
import com.ib.client.EClientErrors.CodeMsgPair;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EReader_stop_12_3_Test {

    @Mock
    private DataInputStream m_dis;

    @Mock
    private EWrapper m_eWrapper;

    @InjectMocks
    private EReader eReader;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    private boolean getPrivateField(EReader eReader, String fieldName) {
        try {
            java.lang.reflect.Field field = EReader.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.getBoolean(eReader);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testStop() throws IOException {
        // Arrange
        when(m_dis.readInt()).thenThrow(new IOException());
        // Act
        eReader.stop();
        // Assert
        assertFalse(getPrivateField(eReader, "m_isSocketOK"));
        assertFalse(getPrivateField(eReader, "m_isStopped"));
        verify(m_dis, times(1)).close();
        verify(m_eWrapper, times(1)).error(NO_VALID_ID, UNKNOWN_ID.code(), UNKNOWN_ID.msg());
    }
}
