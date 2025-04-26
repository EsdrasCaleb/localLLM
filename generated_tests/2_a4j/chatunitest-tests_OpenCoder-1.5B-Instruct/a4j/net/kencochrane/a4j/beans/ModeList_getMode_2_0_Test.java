// Test method
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class ModeList_getMode_2_0_Test {

    @Test
    void testGetMode() {
        ModeList modeList = Mockito.mock(ModeList.class);
        Mode sprintMode = Mockito.mock(Mode.class);
        Mode hybridMode = Mockito.mock(Mode.class);
        Mode backToSprintMode = Mockito.mock(Mode.class);
        Mockito.when(modeList.getMode("Sprint")).thenReturn(sprintMode);
        Mockito.when(modeList.getMode("Hybrid")).thenReturn(hybridMode);
        Mockito.when(modeList.getMode("Back-to-Sprint")).thenReturn(backToSprintMode);
        Mockito.when(modeList.getMode("Field-Test")).thenReturn(null);
        assertEquals("Sprint", Mockito.doReturn("Sprint").when(sprintMode).getModeName());
        assertEquals("Hybrid", Mockito.doReturn("Hybrid").when(hybridMode).getModeName());
        assertEquals("Back-to-Sprint", Mockito.doReturn("Back-to-Sprint").when(backToSprintMode).getModeName());
        assertNull(Mockito.doReturn(null).when(modeList).getMode("Field-Test"));
        // Buggy lines
        assertEquals("Sprint", sprintMode.getModeName());
        assertEquals("Hybrid", hybridMode.getModeName());
        assertEquals("Back-to-Sprint", backToSprintMode.getModeName());
        assertNull(Mockito.doReturn(null).when(modeList).getMode("Field-Test"));
    }
}
