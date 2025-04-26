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

    @Test
    public void testGetMode() throws Exception {
        ModeList modeList = new ModeList();
        Mode mode = new Mode();
        mode.setModeName("test");
        modeList.modes = new ArrayList<>();
        modeList.modes.add(mode);
        Field field = ModeList.class.getDeclaredField("modes");
        field.setAccessible(true);
        field.set(modeList, new ArrayList<>());
        Mode returnedMode = modeList.getMode("test");
        assertEquals(mode, returnedMode);
        returnedMode = modeList.getMode("wrongName");
        assertEquals(null, returnedMode);
        modeList.modes = null;
        returnedMode = modeList.getMode("test");
        assertEquals(null, returnedMode);
    }
}
