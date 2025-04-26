package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ModeList_getMode_2_0_Test {

    @Mock
    private ArrayList<Mode> modes;

    @InjectMocks
    private ModeList modeList;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
        try {
            Field modesField = ModeList.class.getDeclaredField("modes");
            modesField.setAccessible(true);
            modesField.set(modeList, modes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testGetMode_WhenModesListIsNotEmpty_ShouldReturnMode() {
        Mode mode = new Mode();
        mode.setModeName("TestMode");
        when(modes.size()).thenReturn(1);
        when(modes.get(0)).thenReturn(mode);
        Mode result = modeList.getMode("TestMode");
        assertEquals(mode, result);
    }

    @Test
    public void testGetMode_WhenModesListIsEmpty_ShouldReturnNull() {
        when(modes.size()).thenReturn(0);
        Mode result = modeList.getMode("TestMode");
        assertNull(result);
    }

    @Test
    public void testGetMode_WhenModeNameDoesNotExist_ShouldReturnNull() {
        Mode mode = new Mode();
        mode.setModeName("TestMode");
        when(modes.size()).thenReturn(1);
        when(modes.get(0)).thenReturn(mode);
        Mode result = modeList.getMode("NonExistentMode");
        assertNull(result);
    }
}
