package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ModeList_getMode_2_0_Test {

    @InjectMocks
    private ModeList modeList;

    @Mock
    private Mode mode1;

    @Mock
    private Mode mode2;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        modeList = new ModeList();
        modeList.modes = new ArrayList<>();
    }

    @Test
    void testGetModeFound() {
        when(mode1.getModeName()).thenReturn("Mode1");
        when(mode2.getModeName()).thenReturn("Mode2");
        modeList.modes.add(mode1);
        modeList.modes.add(mode2);
        Mode result = modeList.getMode("Mode1");
        assertNotNull(result);
        assertEquals(mode1, result);
    }

    @Test
    void testGetModeNotFound() {
        when(mode1.getModeName()).thenReturn("Mode1");
        when(mode2.getModeName()).thenReturn("Mode2");
        modeList.modes.add(mode1);
        modeList.modes.add(mode2);
        Mode result = modeList.getMode("Mode3");
        assertNull(result);
    }

    @Test
    void testGetModeEmptyList() {
        Mode result = modeList.getMode("Mode1");
        assertNull(result);
    }

    @Test
    void testGetModeNullList() {
        modeList.modes = null;
        Mode result = modeList.getMode("Mode1");
        assertNull(result);
    }
}
