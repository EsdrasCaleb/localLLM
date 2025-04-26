package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ModeList_getMode_2_1_Test {

    private ModeList modeList;

    private Mode mockMode;

    @BeforeEach
    public void setUp() {
        modeList = new ModeList();
        mockMode = mock(Mode.class);
        when(mockMode.getModeName()).thenReturn("Mock Mode");
        modeList.modes.add(mockMode);
    }

    @Test
    public void testGetModeWithExistingMode() {
        String modeName = "Mock Mode";
        Mode result = modeList.getMode(modeName);
        assertNotNull(result);
        assertEquals(mockMode, result);
    }

    @Test
    public void testGetModeWithNonExistentMode() {
        String modeName = "Unknown Mode";
        Mode result = modeList.getMode(modeName);
        assertNull(result);
    }

    @Test
    public void testGetModeWhenModesIsEmpty() {
        modeList.modes.clear();
        String modeName = "Mock Mode";
        Mode result = modeList.getMode(modeName);
        assertNull(result);
    }
}
