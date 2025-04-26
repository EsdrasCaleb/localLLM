package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ModeList_addMode_1_0_Test {

    @Test
    public void testAddMode() throws Exception {
        ModeList modeList = new ModeList();
        Mode mode = Mockito.mock(Mode.class);
        Method addModeMethod = modeList.getClass().getDeclaredMethod("addMode", Mode.class);
        addModeMethod.setAccessible(true);
        addModeMethod.invoke(modeList, mode);
        Assertions.assertEquals(modeList.getAllModes().size(), 1);
    }
}
